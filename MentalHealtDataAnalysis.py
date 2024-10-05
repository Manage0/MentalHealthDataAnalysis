import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Load the CSV into a DataFrame
df = pd.read_csv('MentalHealthSurvey.csv')

def basicInfo():
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

def makePlots():
    # Set the aesthetic style of the plots
    sns.set_theme(style="whitegrid")

    # Get the list of columns and remove 'stress_relief_activities'
    columns = df.columns[df.columns != 'stress_relief_activities']

    # Determine the number of rows and columns for subplots
    n_cols = 2  # Define the number of columns for the grid
    n_rows = math.ceil(len(columns) / n_cols)  # Calculate the number of rows needed

    # Create the subplots grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))  # Increase figure height as needed
    axes = axes.flatten()  # Flatten in case we have a multi-dimensional array of axes

    # Loop through each column and create a plot in the corresponding subplot
    for i, col in enumerate(columns):
        if df[col].dtype == 'object':  # Categorical columns
            sns.countplot(y=col, data=df, order=df[col].value_counts().index, ax=axes[i])  # Bar plot for categorical data
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel('Count')
            axes[i].set_ylabel(col)
            
        elif df[col].dtype in ['int64', 'float64']:  # Numeric columns
            sns.histplot(df[col], kde=True, ax=axes[i])  # Histogram with KDE for numeric columns
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')

    # Adjust layout so that plots don't overlap
    plt.tight_layout()

    # Save the figure as an image for external viewing
    fig.savefig('MentalHealthSurvey_Distribution.png', dpi=300, bbox_inches='tight')  # Save as PNG

    # Extract the 'stress_relief_activities' column
    stress_relief_activities = df['stress_relief_activities']

    # Split activities by commas and flatten the list
    individual_activities_list = []
    for activities in stress_relief_activities.dropna():  # Drop missing values
        individual_activities = [activity.strip() for activity in activities.split(',')]  # Split and remove extra spaces
        individual_activities_list.extend(individual_activities)  # Collect all individual activities

    # Convert the list to a pandas Series for easy counting
    individual_activities_series = pd.Series(individual_activities_list)

    # Create a plot specifically for the unique stress relief activities
    plt.figure(figsize=(10, 6))
    sns.countplot(y=individual_activities_series, order=individual_activities_series.value_counts().index)
    plt.title('Distribution of Individual Stress Relief Activities')
    plt.xlabel('Count')
    plt.ylabel('Stress Relief Activities')

    # Save the plot as a PNG file
    plt.savefig('stress_relief_activities_distribution_unique.png', dpi=300, bbox_inches='tight')
    print("Figures saved as 'stress_relief_activities_distribution_unique.png'")

# Only call the functions you need for trying out. Comment out the ones you don't need
# basicInfo()
makePlots()