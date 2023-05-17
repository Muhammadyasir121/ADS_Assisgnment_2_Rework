# -*- coding: utf-8 -*-
"""Copy of 22030978

## ***Question 1:Data Ingestion and Manipulation using Pandas***
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import pandas as pd

def read_worldbank_data(filename):
    """
    Reads World Bank data from a CSV file and performs data transformation.

    Args:
        filename (str): The path or URL of the CSV file.

    Returns:
        tuple: A tuple containing two pandas dataframes:
            - df_transposed: A dataframe with years as columns and countries as rows.
            - df: The original dataframe with years as rows and countries as columns.
    """

    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(filename, skiprows=4)

    # Drop any empty columns
    df.dropna(axis=1, how='all', inplace=True)

    # Transpose the dataframe to have years as columns
    df_transposed = df.transpose()

    # Clean the transposed dataframe
    new_header = df_transposed.iloc[0]
    df_transposed = df_transposed[1:]
    df_transposed.columns = new_header
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'Country'}, inplace=True)
    df_transposed.set_index('Country', inplace=True)

    # Return dataframes with years as columns and countries as columns
    return df_transposed, df

# Test the function
co2_file = 'https://raw.githubusercontent.com/amna-sarwar/copy2/main/API_EN.ATM.CO2E.PC_DS2_en_csv_v2_5455265.csv'
renewable_file = 'https://raw.githubusercontent.com/amna-sarwar/copy2/main/API_EG.FEC.RNEW.ZS_DS2_en_csv_v2_5457050.csv'

co2_df_years, co2_df_countries = read_worldbank_data(co2_file)
renewable_df_years, renewable_df_countries = read_worldbank_data(renewable_file)

print("CO2 Emissions per Capita - Years as Columns:")
print(co2_df_years.head())

print("\nCO2 Emissions per Capita - Countries as Columns:")
print(co2_df_countries.head())

print("\nRenewable Energy Consumption - Years as Columns:")
print(renewable_df_years.head())

print("\nRenewable Energy Consumption - Countries as Columns:")
print(renewable_df_countries.head())

"""## ***Question 2:Statistical Analysis of CO2 Emissions and Renewable Energy Consumption***"""

# Merge dataframes based on 'Country Name' column

merged_df_countries = pd.merge(co2_df_countries, renewable_df_countries, on='Country Name')

# Get the column names for years
year_columns = merged_df_countries.columns[4:].tolist()  # Get the column names for years

# Convert CO2 emissions values to numeric
co2_values = pd.to_numeric(merged_df_countries[year_columns].stack(), errors='coerce').values
# Convert renewable energy values to numeric
renewable_values = pd.to_numeric(merged_df_countries[year_columns].stack(), errors='coerce').values

"""### ***Summary Statistics:***"""

# Calculate summary statistics for CO2 emissions per capita
co2_summary = merged_df_countries[year_columns].describe()
# Calculate summary statistics for renewable energy consumption
renewable_summary = merged_df_countries[year_columns].describe()

print("CO2 Emissions per Capita - Summary Statistics:")
print(co2_summary)

print("\nRenewable Energy Consumption - Summary Statistics:")
print(renewable_summary)

"""### ***Scatter Plot:***"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get the available range of years
start = co2_df_years.columns[0]
end = co2_df_years.columns[-1]

# Subset the data for the available range of years
co2_subset = co2_df_years.loc[:, start:end]
renewable_subset = renewable_df_years.loc[:, start:end]

# Convert data to numeric values
co2_values = pd.to_numeric(co2_subset.values.flatten(), errors='coerce')
renewable_values = pd.to_numeric(renewable_subset.values.flatten(), errors='coerce')

# Scatter plot
plt.scatter(co2_values, renewable_values)
plt.xlabel("CO2 Emissions per Capita")
plt.ylabel("Renewable Energy Consumption")
plt.title("Scatter Plot ({}-{})".format(start, end))
plt.show()

"""### ***Correlation Coefficient:***"""

# Calculate correlation coefficient between CO2 emissions per capita and renewable energy consumption

# Select valid indices where both CO2 and renewable values are finite
valid_indices = np.logical_and(np.isfinite(co2_values), np.isfinite(renewable_values))

# Filter CO2 and renewable values using valid indices
co2_valid = co2_values[valid_indices]
renewable_valid = renewable_values[valid_indices]

# Center the data by subtracting the means
x = co2_valid - np.mean(co2_valid)
y = renewable_valid - np.mean(renewable_valid)

# Calculate correlation coefficient
correlation_coefficient = np.sum(x * y) / np.sqrt(np.sum(x**2) * np.sum(y**2))

# Print the correlation coefficient
print("Correlation Coefficient: {:.2f}".format(correlation_coefficient))

"""### ***T-Statistic and P-Value:***"""

# Perform independent two-sample t-test

# Select valid indices where both CO2 and renewable values are finite
valid_indices = np.logical_and(np.isfinite(co2_values), np.isfinite(renewable_values))

# Filter CO2 and renewable values using valid indices
co2_valid = co2_values[valid_indices]
renewable_valid = renewable_values[valid_indices]

# Calculate sample sizes
n1 = len(co2_valid)
n2 = len(renewable_valid)

# Calculate sample means
mean1 = np.mean(co2_valid)
mean2 = np.mean(renewable_valid)

# Calculate sample variances
var1 = np.var(co2_valid, ddof=1)
var2 = np.var(renewable_valid, ddof=1)

# Calculate pooled variance
pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

# Calculate t-statistic
t_statistic = (mean1 - mean2) / np.sqrt(pooled_var * (1/n1 + 1/n2))

# Calculate degrees of freedom
dof = n1 + n2 - 2

# Calculate p-value
p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), dof))

# Print the t-statistic and p-value
print("T-Statistic: {:.2f}".format(t_statistic))
print("P-Value: {:.2f}".format(p_value))

# Visualize statistical analysis results

# Create a horizontal bar plot
plt.barh(['Correlation Coefficient', 'T-Statistic', 'P-Value'], [correlation_coefficient, t_statistic, p_value], color=['blue', 'green', 'red'])

# Set the y-axis label and plot title
plt.ylabel('Value')
plt.title('Statistical Analysis Results')

# Display the plot
plt.show()

"""## ***Question 3:Exploring Correlations between Indicators and Their Variation Over Time***"""

# Visualize correlation matrix of CO2 emissions and renewable energy consumption

# Define the desired range of years
start_year = 2017
end_year = 2019

# Subset the CO2 emissions data for the desired range of years
co2_subset = co2_df_countries[['Country Name'] + [str(year) for year in range(start_year, end_year + 1)]]

# Subset the renewable energy consumption data for the desired range of years
renewable_subset = renewable_df_countries[['Country Name'] + [str(year) for year in range(start_year, end_year + 1)]]

# Fill missing values with 0 in the subsets
co2_subset = co2_subset.fillna(0)
renewable_subset = renewable_subset.fillna(0)

# Merge the CO2 emissions and renewable energy consumption data on 'Country Name'
merged_data = pd.merge(co2_subset, renewable_subset, on='Country Name')

# Rename the columns to remove suffixes
merged_data.columns = merged_data.columns.str.replace('_x', '')
merged_data.columns = merged_data.columns.str.replace('_y', '')

# Compute the correlation matrix
correlation_matrix = merged_data.iloc[:, 1:].corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)

# Add text annotations for correlation values
for i in range(len(merged_data.columns[1:])):
    for j in range(len(merged_data.columns[1:])):
        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='white')

plt.colorbar()
plt.xticks(range(len(merged_data.columns[1:])), merged_data.columns[1:], rotation=90)
plt.yticks(range(len(merged_data.columns[1:])), merged_data.columns[1:])
plt.xlabel('Year')
plt.ylabel('Year')
plt.title('Correlation Matrix: CO2 Emissions vs. Renewable Energy Consumption')
plt.show()

"""## ***Question 4:Trends in CO2 Emissions and Renewable Energy Consumption***"""

# Visualize trends in CO2 emissions and renewable energy consumption

# Read CO2 emissions dataset
co2_file = 'https://raw.githubusercontent.com/amna-sarwar/copy2/main/API_EN.ATM.CO2E.PC_DS2_en_csv_v2_5455265.csv'
co2_df = pd.read_csv(co2_file, skiprows=4)

# Read renewable energy consumption dataset
renewable_file = 'https://raw.githubusercontent.com/amna-sarwar/copy2/main/API_EG.FEC.RNEW.ZS_DS2_en_csv_v2_5457050.csv'
renewable_df = pd.read_csv(renewable_file, skiprows=4)

# Subset the data for the years 2001 to 2019
co2_subset = co2_df.loc[:, '2001':'2019']
renewable_subset = renewable_df.loc[:, '2001':'2019']

# Calculate the average for each year
co2_avg = co2_subset.mean()
renewable_avg = renewable_subset.mean()

# Plot the trends
plt.figure(figsize=(12, 6)) # Increase the width of the figure
plt.plot(co2_avg.index, co2_avg.values, label='CO2 Emissions per Capita')
plt.plot(renewable_avg.index, renewable_avg.values, label='Renewable Energy Consumption')
plt.xlabel('Year')
plt.ylabel('Average')
plt.title('Trends in CO2 Emissions and Renewable Energy Consumption (2001-2019)')
plt.legend()
plt.show()

# Perform seasonal decomposition of CO2 emissions per capita
from statsmodels.tsa.seasonal import seasonal_decompose
# Subset the data for the years 1990 to 2019
co2_subset = co2_df.loc[:, '1990':'2019']

# Calculate the average for each year
co2_avg = co2_subset.mean()

# Perform seasonal decomposition for CO2 emissions
co2_decomposition = seasonal_decompose(co2_avg, model='additive', period=12)

# Plot the decomposed components for CO2 emissions
plt.figure(figsize=(14, 12))
plt.subplot(411)
plt.plot(co2_decomposition.observed)
plt.ylabel('Observed')
plt.subplot(412)
plt.plot(co2_decomposition.trend)
plt.ylabel('Trend')
plt.subplot(413)
plt.plot(co2_decomposition.seasonal)
plt.ylabel('Seasonal')
plt.subplot(414)
plt.plot(co2_decomposition.resid)
plt.ylabel('Residual')
plt.xlabel('Year')
plt.suptitle('Seasonal Decomposition of CO2 Emissions per Capita')
plt.tight_layout()
plt.show()