import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as stats
import numpy as np
import statsmodels.api as sm

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
    for activities in stress_relief_activities:
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

def isGenderAndSportsRelated():
    # Crosstab between gender and sports_engagement
    contingency_table = pd.crosstab(df['gender'], df['sports_engagement'])

    # Chi-squared test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    print(f"Chi-squared Statistic: {chi2}")
    print(f"P-value: {p}")
    print(f"Degrees of Freedom: {dof}")
    print(f"Expected Frequencies: \n{expected}")

    # If p < 0.05, we reject the null hypothesis (there is an association)
    if p < 0.05:
        print("There is a statistically significant association between gender and sports engagement.")
    else:
        print("No statistically significant association between gender and sports engagement.")

def convertSleepToNumeric(sleep_data):
    """
    Converts sleep data from string ranges to their numeric midpoints.
    Example: '4-6 hrs' -> 5.0
    """
    numeric_sleep = []
    for entry in sleep_data:
        if '-' in entry:
            # Split the range and compute the midpoint
            lower, upper = entry.replace('hrs', '').split('-')
            midpoint = (float(lower) + float(upper)) / 2
            numeric_sleep.append(midpoint)
        else:
            # If it's a single value (e.g., '6 hrs'), just take that value
            numeric_sleep.append(float(entry.replace('hrs', '').strip()))
    return np.array(numeric_sleep)

def compareSleepByGender():
    """
    This function compares the distribution of sleep hours categories between males and females 
    using the chi-square test of independence.
    
    Returns:
    None: Prints the chi-square statistic, p-value, and the interpretation of the result.
    """
    
    # Create a contingency table for gender and average_sleep
    contingency_table = pd.crosstab(df['gender'], df['average_sleep'])
    
    # Perform chi-square test
    chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)

    print(f"Chi-square Statistic: {chi2_stat}")
    print(f"P-value: {p_val}")

    if p_val < 0.05:
        print("There is a statistically significant difference in sleep hour categories between males and females.")
    else:
        print("No statistically significant difference in sleep hour categories between males and females.")

def analyzeAcademicPressureDepressionCorrelation(): 
    """
    This function calculates the Kendall Tau correlation between academic pressure 
    and depression and checks for statistical significance.
    
    Returns:
    None: Prints the Kendall Tau correlation coefficient, p-value, and the interpretation of the result.
    """
    
    # Convert relevant columns to numeric
    academic_pressure = pd.to_numeric(df['academic_pressure'], errors='coerce')
    depression = pd.to_numeric(df['depression'], errors='coerce')

    # Kendall Tau correlation
    correlation, p_value = stats.kendalltau(academic_pressure, depression)

    print(f"Kendall Tau Correlation: {correlation}")
    print(f"P-value: {p_value}")

    if p_value < 0.05:
        print("There is a statistically significant correlation between academic pressure and depression.")
    else:
        print("No statistically significant correlation between academic pressure and depression.")

def compareSleepAcrossAcademicYears():
    """
    This function compares average sleep hours across different academic years 
    using an ANOVA test to check for statistically significant differences.
    
    Returns:
    None: Prints the F-statistic, p-value, and the interpretation of the result.
    """

    # Convert 'average_sleep' column to numeric values
    df['average_sleep_numeric'] = convertSleepToNumeric(df['average_sleep'])
    
    # Group the data by academic year and extract the numeric sleep data
    grouped_data = [df[df['academic_year'] == year]['average_sleep_numeric']
                    for year in df['academic_year'].unique()]

    # ANOVA test
    f_stat, p_val = stats.f_oneway(*grouped_data)

    print(f"F-statistic: {f_stat}")
    print(f"P-value: {p_val}")

    if p_val < 0.05:
        print("There is a statistically significant difference in sleep hours across academic years.")
    else:
        print("No statistically significant difference in sleep hours across academic years.")

def performLogisticRegression():
    """
    This function performs logistic regression to analyze the relationship 
    between academic pressure, financial concerns, and depression.
    
    Returns:
    None: Prints the summary of the logistic regression model.
    """

    # Recode 'depression' into a binary variable
    threshold = 3  # Set threshold as needed
    df['depression_binary'] = df['depression'].apply(lambda x: 1 if x >= threshold else 0)


    # Convert relevant columns to numeric
    df['depression_binary'] = pd.to_numeric(df['depression_binary'], errors='coerce')
    df['academic_pressure'] = pd.to_numeric(df['academic_pressure'], errors='coerce')
    df['financial_concerns'] = pd.to_numeric(df['financial_concerns'], errors='coerce')

    # Select independent variables
    X = df[['academic_pressure', 'financial_concerns']]
    y = df['depression_binary']

    # Add constant to the model (for intercept)
    X = sm.add_constant(X)

    # Fit the logistic regression model
    model = sm.Logit(y, X)
    result = model.fit()

    # Print summary
    print(result.summary())

    # Note: A logisztikus regresszió eredményei azt mutatják, hogy mind az akadémiai nyomás, mind a pénzügyi aggodalmak jelentős hatással vannak a depresszióra.

# Only call the functions you need for trying out. Comment out the ones you don't need
# basicInfo()
# makePlots()
# isGenderAndSportsRelated()
# compareSleepByGender()
analyzeAcademicPressureDepressionCorrelation()
# compareSleepAcrossAcademicYears()
# performLogisticRegression()