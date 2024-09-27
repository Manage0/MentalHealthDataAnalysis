import pandas as pd

# Load the CSV into a DataFrame
df = pd.read_csv('MentalHealthSurvey.csv')

# Show the first few rows to inspect the data
print("First 5 rows of the dataset:")
print(df.head())

# Get basic statistics and information
print("\nBasic Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Display value counts for categorical columns like gender
print("\nValue counts for gender:")
print(df['gender'].value_counts())

# Check for missing data
print("\nCheck for missing data:")
print(df.isnull().sum())
