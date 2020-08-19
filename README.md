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
customers.describe()
customers.info()
```
## Exploratory Data Analysis

**Let's explore the data!**
```python
sns.jointplot('Time on Website', 'Yearly Amount Spent', data=customers)
```
![alt text](https://github.com/yash-kh/Linear-Regression-Project/Plots/TOM-YAS.jpg?raw=true)




![alt text](https://github.com/yash-kh/Linear-Regression-Project/blob/image.jpg?raw=true)
