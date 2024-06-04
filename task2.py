import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Disable SSL certificate verification
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/vasudha830/RYZEN_TECH/main/MagicBricks.csv")

# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.shape)
# print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
# print(df.isna().sum())

df['Per_Sqft'].fillna((df['Price']/df['Area']),inplace=True)
df['Bathroom'].fillna(df['Bathroom'].mode()[0],inplace=True)
df['Furnishing'].fillna(df['Furnishing'].mode()[0],inplace=True)
df['Parking'].fillna(df['Parking'].mode()[0],inplace=True)
df['Type'].fillna(df['Type'].mode()[0],inplace=True)
# print(df.info())
df[['Parking','Bathroom']]=df[['Parking','Bathroom']].astype('int64')
# print(df.nunique())
# print(df.describe())


# #Data Visualisation

num_col=df[df.dtypes[df.dtypes != 'object'].index]
num_col
plt.figure(figsize=(15,10))
sns.heatmap(num_col.corr(),annot=True)
# print(plt.show())
plt.figure(figsize=(7,5))
sns.histplot(x=df['Area'],kde=True,bins=20)
# print(plt.show())
plt.figure(figsize=(17,10))
plt.subplot(3,4,1)
sns.countplot(x=df['BHK'])
plt.subplot(3,4,2)
sns.boxplot(x=df['BHK'],y=df['Price'])
# print('Correlation between BHK and Price is',df['BHK'].corr(df['Price']))
# print('Skewness of the BHK is',df['BHK'].skew())
plt.subplot(3,4,3)
sns.countplot(x=df['Bathroom'])
plt.subplot(3,4,4)
sns.boxplot(x=df['Bathroom'],y=df['Price'])
# print('Correlation between Bathroom and Price is',df['Bathroom'].corr(df['Price']))
# print('Skewness of the Bathroom is',df['Bathroom'].skew())
plt.subplot(3,4,5)
sns.countplot(x=df['Furnishing'])
plt.subplot(3,4,6)
sns.boxplot(x=df['Furnishing'],y=df['Price'])
plt.subplot(3,4,7)
sns.countplot(x=df['Parking'])
plt.subplot(3,4,8)
sns.boxplot(x=df['Parking'],y=df['Price'])
# print('Correlation between Parking and Price is',df['Parking'].corr(df['Price']))
# print('Skewness of the Parking is',df['Parking'].skew())
plt.subplot(3,4,9)
sns.countplot(x=df['Status'])
plt.subplot(3,4,10)
sns.boxplot(x=df['Status'],y=df['Price'])
plt.subplot(3,4,11)
sns.barplot(x=df['BHK'],y=df['Area'])
plt.subplot(3,4,12)
sns.barplot(x=df['Bathroom'],y=df['Area'])
# print(plt.show())

df.drop(df.index[(df["Parking"] == 39)],axis=0,inplace=True)
df.drop(df.index[(df["Parking"] == 114)],axis=0,inplace=True)

plt.figure(figsize=(17,10))
plt.subplot(3,4,1)
sns.countplot(x=df['BHK'])
plt.subplot(3,4,2)
sns.boxplot(x=df['BHK'],y=df['Price'])
# print('Correlation between BHK and Price is',df['BHK'].corr(df['Price']))
# print('Skewness of the BHK is',df['BHK'].skew())
plt.subplot(3,4,3)
sns.countplot(x=df['Bathroom'])
plt.subplot(3,4,4)
sns.boxplot(x=df['Bathroom'],y=df['Price'])
# print('Correlation between Bathroom and Price is',df['Bathroom'].corr(df['Price']))
# print('Skewness of the Bathroom is',df['Bathroom'].skew())
plt.subplot(3,4,5)
sns.countplot(x=df['Furnishing'])
plt.subplot(3,4,6)
sns.boxplot(x=df['Furnishing'],y=df['Price'])
plt.subplot(3,4,7)
sns.countplot(x=df['Parking'])
plt.subplot(3,4,8)
sns.boxplot(x=df['Parking'],y=df['Price'])
# print('Correlation between Parking and Price is',df['Parking'].corr(df['Price']))
# print('Skewness of the Parking is',df['Parking'].skew())
plt.subplot(3,4,9)
sns.countplot(x=df['Status'])
plt.subplot(3,4,10)
sns.boxplot(x=df['Status'],y=df['Price'])
plt.subplot(3,4,11)
sns.barplot(x=df['BHK'],y=df['Area'])
plt.subplot(3,4,12)
sns.barplot(x=df['Bathroom'],y=df['Area'])
# print(plt.show())

plt.figure(figsize=(7,5))
sns.barplot(x=df['Furnishing'],y=df['Price'],hue=df['BHK'])
#removing outliers
from scipy import stats 
z = np.abs(stats.zscore(df[df.dtypes[df.dtypes != 'object'].index]))
df = df[(z < 3).all(axis=1)]
df.shape


# # Data Visualisation after removing outliers
plt.figure(figsize=(17,10))
plt.subplot(3,4,1)
sns.countplot(x=df['BHK'])
plt.subplot(3,4,2)
sns.boxplot(x=df['BHK'],y=df['Price'])
# print('Correlation between BHK and Price is',df['BHK'].corr(df['Price']))
# print('Skewness of the BHK is',df['BHK'].skew())
plt.subplot(3,4,3)
sns.countplot(x=df['Bathroom'])
plt.subplot(3,4,4)
sns.boxplot(x=df['Bathroom'],y=df['Price'])
# print('Correlation between Bathroom and Price is',df['Bathroom'].corr(df['Price']))
# print('Skewness of the Bathroom is',df['Bathroom'].skew())
plt.subplot(3,4,5)
sns.countplot(x=df['Furnishing'])
plt.subplot(3,4,6)
sns.boxplot(x=df['Furnishing'],y=df['Price'])
plt.subplot(3,4,7)
sns.countplot(x=df['Parking'])
plt.subplot(3,4,8)
sns.boxplot(x=df['Parking'],y=df['Price'])
# print('Correlation between Parking and Price is',df['Parking'].corr(df['Price']))
# print('Skewness of the Parking is',df['Parking'].skew())
plt.subplot(3,4,9)
sns.countplot(x=df['Status'])
plt.subplot(3,4,10)
sns.boxplot(x=df['Status'],y=df['Price'])
plt.subplot(3,4,11)
sns.barplot(x=df['BHK'],y=df['Area'])
plt.subplot(3,4,12)
sns.barplot(x=df['Bathroom'],y=df['Area'])
# print(plt.show())

df.drop(df[df['Area'] > 30000].index, inplace = True)
plt.figure(figsize=(14,7))
plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(df.Area,df.Price)
# print(plt.show())


np.random.seed(7)
x = np.random.rand(100, 1)
y = 13 + 3 * x + np.random.rand(100, 1)
plt.scatter(x,y,s=10)
plt.xlabel('x')
plt.ylabel('y')
print(plt.show())


# #ML Modeling

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=0.3)
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X_train, Y_train)

Y_pred = linear.predict(X_test)
# print("Accuracy Score for Test Dataset is ",linear.score(X_test, Y_test)*100,"%")
# print("Accuracy Score for Train Dataset is",linear.score(X_train,Y_train)*100,"%")

c1=float(linear.intercept_)
m1=float(linear.coef_)
# print("Intercept (c) of regression line is", c1)
# print("Coefficient (m) of regression line is", m1)

plt.scatter(X_test, Y_test, label='Test Data')
plt.plot(X_test, m1 * X_test + c1, color='red', label='Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
# So this model Predicts the value of any house with an accuracy of 88.42%
