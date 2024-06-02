import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib import pyplot
from datetime import datetime
url = 'https://raw.githubusercontent.com/vasudha830/RYZEN_TECH/main/fifa21%20raw%20data%20v2.csv'
data = pd.read_csv(url, low_memory=False)
# print(data.head())

data.isnull().sum()
pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None) 
data_types = data.dtypes
# print(data_types)

# function to convert height to inches
def height_to_inches(height):
    try:
        if 'cm' in height:
            cm = int(height.replace('cm', ''))
            inches = cm / 2.54  # Convert cm to inches
            return inches
        else:
            feet, inches = height.split("'")
            feet = int(feet)
            inches = int(inches.replace('"', ''))
            total_inches = feet * 12 + inches
            return total_inches
    except ValueError as e:
        print(f"Error processing height: {height}, error: {e}")
        return None

# Function to convert weight from lbs to pounds
def weight_to_pounds(weight):
    try:
        if 'lbs' in weight:
            return int(weight.replace('lbs', ''))
        elif 'kg' in weight:
            kg = int(weight.replace('kg', ''))
            return int(kg * 2.20462)  # Convert kg to lbs
        else:
            raise ValueError("Unknown weight format")
    except ValueError as e:
        print(f"Error processing weight: {weight}, error: {e}")
        return None

# Apply this function to the Height column
data['Height'] = data['Height'].apply(height_to_inches)
data['Weight'] = data['Weight'].apply(weight_to_pounds)
for column in data.select_dtypes(include=[object]):
    data[column] = data[column].str.replace('\n', '', regex=False)

# Based on the 'Joined' column, checking which players have been playing at a club for more than 10 years!
data['Joined'] = pd.to_datetime(data['Joined'], format='%b %d, %Y')
data['Years_at_club'] = (datetime.now() - data['Joined']).dt.days // 365
veterans = data[data['Years_at_club'] > 10]
# print (veterans['Name'])

# 'Value', 'Wage' and "Release Clause' are string columns. Converting them to numbers. For eg, "M" in value column is Million, so multiply the row values by 1,000,000, etc.
def convert_financials(s):
    # Remove the currency symbol
    s = s.replace('€', '')

    # Convert millions 'M' to a number
    if 'M' in s:
        return float(s.replace('M', '')) * 1e6

    # Convert thousands 'K' to a number
    elif 'K' in s:
        return float(s.replace('K', '')) * 1e3

    # Return the value as it is
    return float(s)

# Apply the function to each of the financial columns
data['Value'] = data['Value'].apply(convert_financials)
data['Wage'] = data['Wage'].apply(convert_financials)
data['Release Clause'] = data['Release Clause'].apply(convert_financials)
# print(data.head())

# Stripping those columns of these stars and making the columns numerical
data['SM'] = data['SM'].str.replace('★', '')
data['W/F'] = data['W/F'].str.replace('\n', '')
# print(data.head())

# Remove asterisks from 'W/F' and 'IR' columns
data['W/F'] = data['W/F'].str.replace('★', '')
data['IR'] = data['IR'].str.replace('★', '')

# Which players are highly valuable but still underpaid (on low wages)? (hint: scatter plot between wage and value)
plt.figure(figsize=(10, 6))
plt.scatter(data['Value'], data['Wage'], alpha=0.5)
plt.title('Scatter Plot of Player Value vs. Wage')
plt.xlabel('Player Value (€)')
plt.ylabel('Wage (€ per week)')
plt.xscale('log')
plt.yscale('log')
# print(plt.show())

# Dropping photoUrl and playeURL: as those are irrelevant to the dataset.
data = data.drop('photoUrl', axis=1)
data = data.drop('playerUrl', axis=1)
# print(data.head())

# Splitting the columns Team & Contract
idx = data.columns.get_loc('Contract')
# Extract and insert 'Team' and 'New_Contract' columns
data.insert(idx, 'Team', data['Contract'].str.extract(r'([a-zA-Z\s]+)')[0])
data.insert(idx + 1, 'New_Contract', data['Contract'].str.extract(r'(\d{4} ~ \d{4})')[0])
data.drop('Contract', axis=1, inplace=True)
# print(data.head())

idx = data.columns.get_loc('LongName')
print(idx)
columns = ['Name'] + [col for col in data.columns if col != 'Name']
# Reindex the DataFrame columns based on the new order
data = data.reindex(columns=columns)

data.drop('LongName', axis=1, inplace=True)
# print(data.head())

columns = ['ID'] + [col for col in data.columns if col != 'ID']
data = data.reindex(columns=columns)
print(data.head())


