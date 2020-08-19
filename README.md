# Linear-Regression-Project
A Linear Regression project to help a company to decide whether to focus their efforts on their mobile app experience or their website.
# Imports
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
%matplotlib inline
```
# Get the Data

We'll work with the Ecommerce Customers csv file from the company. It has Customer info, suchas Email, Address, and their color Avatar. Then it also has numerical value columns:

* Avg. Session Length: Average session of in-store style advice sessions.
* Time on App: Average time spent on App in minutes
* Time on Website: Average time spent on Website in minutes
* Length of Membership: How many years the customer has been a member.
```python
customers = pd.read_csv('Ecommerce Customers')
```
# Examining the Data
```python
customers.head()
```
![image](https://github.com/yash-kh/Linear-Regression-Project/blob/master/Plots/8.PNG?raw=true)
```
customers.describe()
```
![image](https://github.com/yash-kh/Linear-Regression-Project/blob/master/Plots/9.PNG?raw=true)
```
customers.info()
```
![image](https://github.com/yash-kh/Linear-Regression-Project/blob/master/Plots/10.PNG?raw=true)

## Exploratory Data Analysis

**Let's explore the data!**
```python
sns.jointplot('Time on Website', 'Yearly Amount Spent', data=customers)
```
![image](https://raw.githubusercontent.com/yash-kh/Linear-Regression-Project/master/Plots/TOM-YAS.png)
```python
sns.jointplot('Time on App', 'Yearly Amount Spent', data=customers)
```
![image](https://github.com/yash-kh/Linear-Regression-Project/blob/master/Plots/2.png?raw=true)
```python
sns.jointplot('Time on App', 'Length of Membership', data=customers, kind='hex')
```
![image](https://github.com/yash-kh/Linear-Regression-Project/blob/master/Plots/3.png?raw=true)
```python
sns.pairplot(customers)
```
![image](https://github.com/yash-kh/Linear-Regression-Project/blob/master/Plots/4.png?raw=true)

 **Based off this plot Length of Membership looks to be the most correlated feature with Yearly Amount Spent**
 ```python
 sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)
 ```
 ![image](https://github.com/yash-kh/Linear-Regression-Project/blob/master/Plots/5.png?raw=true)
 # Training and Testing Data

Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.
```python
y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```
# Training the Model

Now its time to train our model on our training data!
```python
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
```
# The coefficients
```python
print('Coefficients: \n', lm.coef_)
```
![image](https://github.com/yash-kh/Linear-Regression-Project/blob/master/Plots/11.PNG?raw=true)
# Predicting Test Data
Now that we have fit our model, let's evaluate its performance by predicting off the test values!
```python
predictions = lm.predict( X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
```
![image](https://github.com/yash-kh/Linear-Regression-Project/blob/master/Plots/6.png?raw=true)
 **With the plot we can see we made a great linear model because it can fit on a linear line pretty good**
 # Evaluating the Model

Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).
```python
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```
![image](https://github.com/yash-kh/Linear-Regression-Project/blob/master/Plots/12.PNG?raw=true)
# Residuals

You should have gotten a very good model with a good fit. Let's quickly explore the residuals to make sure everything was okay with our data. 
```python
sns.distplot((y_test-predictions),bins=50);
```
![image](https://github.com/yash-kh/Linear-Regression-Project/blob/master/Plots/7.png?raw=true)
# Conclusion
We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important.  Let's see if we can interpret the coefficients at all to get an idea.
```python
coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients
```
![image](https://github.com/yash-kh/Linear-Regression-Project/blob/master/Plots/13.PNG?raw=true)
# Interpreting the coefficients:

- Holding all other features fixed, a 1 unit increase in **Avg. Session Length** is associated with an **increase of 25.98 total dollars spent**.
- Holding all other features fixed, a 1 unit increase in **Time on App** is associated with an **increase of 38.59 total dollars spent**.
- Holding all other features fixed, a 1 unit increase in **Time on Website** is associated with an **increase of 0.19 total dollars spent**.
- Holding all other features fixed, a 1 unit increase in **Length of Membership** is associated with an **increase of 61.27 total dollars spent**.
This is tricky, there are two ways to think about this: Develop the Website to catch up to the performance of the mobile app, or develop the app more since that is what is working better. This sort of answer really depends on the other factors going on at the company, you would probably want to explore the relationship between Length of Membership and the App or the Website before coming to a conclusion!
